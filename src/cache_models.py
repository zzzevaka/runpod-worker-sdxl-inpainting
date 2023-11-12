from utils import get_deployment


if __name__ == "__main__":
    deployment = get_deployment()
    deployment.download_model()
