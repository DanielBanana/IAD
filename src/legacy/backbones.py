from timm.models import list_models


if __name__ == "__main__":
    # list all pretrained models in timm
    for model_name in list_models(pretrained=True):
        print(model_name)
