import shap
import torch
class BiasExplainer:
    def __init__(self, model, tokenizer, device, class_names):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names
        self.model.eval()
        
        # SHAP expects a callable that returns a dict with 'input_ids'
        self.shap_tokenizer = lambda text: {"input_ids": self.tokenizer.encode(text)[0]}

    def predictor(self, texts):
        input_ids = []
        attention_masks = []
        for text in texts:
            ids, mask = self.tokenizer.encode(text)
            input_ids.append(ids)
            attention_masks.append(mask)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs
    
    def explain(self, text):
        explainer = shap.Explainer(
            self.predictor,
            masker=shap.maskers.Text(self.shap_tokenizer),
            output_names=self.class_names
        )
        return explainer([text])
