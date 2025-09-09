import fiftyone.zoo as foz # zoo datasets and models
import fiftyone.core.dataset as fod
import fiftyone.brain as fob # ML methods

def clipEmbedding(dataset:fod.Dataset):
    model = foz.load_zoo_model(
        "clip-vit-base32-torch"
    )  # load the CLIP model from the zoo

    # Compute embeddings for the dataset
    dataset.compute_embeddings(
        model=model, embeddings_field="clip_embeddings", batch_size=64
    )

    # Dimensionality reduction using UMAP on the embeddings
    fob.compute_visualization(
        dataset, embeddings="clip_embeddings", method="umap", brain_key="clip_vis"
    )

def resnetEmbedding(dataset:fod.Dataset):
    model = foz.load_zoo_model(
        "resnet50-imagenet-torch"
    )  # load the ResNet50 model from the zoo

    # Compute embeddings for the dataset — this might take a while on a CPU
    dataset.compute_embeddings(model=model, embeddings_field="resnet50_embeddings")

    # Dimensionality reduction using UMAP on the embeddings
    fob.compute_visualization(
        dataset,
        embeddings="resnet50_embeddings",
        method="umap",
        brain_key="resnet50_vis",
    )