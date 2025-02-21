import tritonclient.grpc as grpcclient
import numpy as np
from torchvision.transforms import Normalize, ToTensor, Resize, Compose
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2


class Osnet:
    def __init__(self, url, model_name="osnet"):
        """
        Initializes the MyModel instance with the Triton server URL and model name.
        """
        self.url = url
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=self.url)

    @property
    def transform(self):
        return Compose(
            [
                ToTensor(),
                Resize((256, 128)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image: np.ndarray):
        """
        Sends a preprocessed image to the Triton server and retrieves embeddings.
        """

        image = self.transform(image).numpy()
        # images = np.array(images)
        # batch = torch.stack([self.transform(image) for image in images])
        # batch = batch.numpy().astype(np.float16)
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float16)

        inputs = []
        outputs = []

        inputs.append(grpcclient.InferInput("input", image.shape, "FP16"))
        inputs[0].set_data_from_numpy(image)
        outputs.append(grpcclient.InferRequestedOutput("output"))

        response = self.client.infer(
            model_name=self.model_name, inputs=inputs, outputs=outputs
        )

        embeddings = response.as_numpy("output")
        return embeddings[0]

    def predict_on_batch(self, images: list, num_workers: int = 8):
        # Prepare a list to hold results in the correct order
        embeddings = [None] * len(images)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks with index to ensure order
            future_to_index = {
                executor.submit(self.predict, image): idx
                for idx, image in enumerate(images)
            }

            # Collect results as tasks complete, and place them in the correct index
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as exc:
                    print(f"Error processing image at index {idx}: {exc}")

        return embeddings


# Example usage
if __name__ == "__main__":
    model = Osnet(url="192.168.2.135:8001", model_name="osnet")
    image_path = "data/reid/a1.jpg"
    img = cv2.imread(image_path)

    import time

    start_time = time.time()

    for i in range(1000):
        embeddings = model.predict_on_batch([img, img, img, img])

    print("Inference time:", (time.time() - start_time) / 1000)
    print("Embeddings shape:", len(embeddings), embeddings[0].shape)

    for i in range(1000):
        embeddings = model.predict(img)

    print("Inference time:", (time.time() - start_time) / 1000)
    print("Embeddings shape:", embeddings.shape)
