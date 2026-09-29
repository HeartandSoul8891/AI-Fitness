import torch
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
import timm
from torchvision import transforms


class WD14Tagger:
    def __init__(self, model_repo="SmilingWolf/wd-v1-4-convnext-tagger-v2"):
        self.model_repo = model_repo
        self.device = self._get_device()
        
        # Preprocessing matching WD14 specifications
        self.preprocess = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        
        # Load weights and metadata
        self.model, self.tags_df = self._load_model_and_tags()

    def _get_device(self) -> torch.device:
        """Detect ROCm/CUDA PyTorch device or fall back to CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model_and_tags(self):
        csv_path = hf_hub_download(repo_id=self.model_repo, filename="selected_tags.csv")
        tags_df = pd.read_csv(csv_path)
        
        model = timm.create_model(f"hf_hub:{self.model_repo}", pretrained=True)
        model.eval()
        model.to(self.device)
        
        return model, tags_df

    def interrogate(self, image: Image.Image, threshold: float = 0.35, char_threshold: float = 0.75):
        """
        Runs inference on a PIL Image.
        Returns:
            tag_string (str): Comma-separated list of tags.
            df_results (pd.DataFrame): DataFrame containing tags and probabilities.
        """
        img_rgb = image.convert("RGB")
        img_tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.sigmoid(self.model(img_tensor))[0].cpu().numpy()

        results_df = self.tags_df.copy()
        results_df["probs"] = probs

        # Filter categories (0 = general, 4 = character)
        general = results_df[(results_df["category"] == 0) & (results_df["probs"] >= threshold)]
        characters = results_df[(results_df["category"] == 4) & (results_df["probs"] >= char_threshold)]

        selected_tags = pd.concat([characters, general])
        tag_list = selected_tags["name"].tolist()
        tag_string = ", ".join(tag_list).replace("_", " ")

        return tag_string, results_df


# Quick CLI test block when running script directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Testing tagger on {img_path}...")
        tagger = WD14Tagger()
        img = Image.open(img_path)
        tags, _ = tagger.interrogate(img)
        print("\nTags:\n", tags)
    else:
        print("Usage: python tagger_engine.py <path_to_image>")