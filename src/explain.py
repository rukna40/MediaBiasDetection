import shap
import torch
class BiasExplainer:
    def __init__(self, model, tokenizer, device, class_names):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names
        self.model.eval()

        # Create full SHAP-compatible tokenizer
        class ShapTokenizer:
            def __init__(self, base_tokenizer):
                self.base = base_tokenizer
                self.mask_token = "[MASK]"

            def __call__(self, text, return_offsets_mapping=False):
                # Tokenize and build offset mapping
                tokens = ['[CLS]'] + self.base.tokenize(text) + ['[SEP]']
                token_ids = [self.base.vocab.get(token, self.base.vocab['[UNK]']) for token in tokens]

                offset_mapping = []
                pos = 0
                text_lower = text.lower() if self.base.lower else text
                for token in tokens:
                    if token in ['[CLS]', '[SEP]']:
                        offset_mapping.append((0, 0))
                    else:
                        # Find token in text
                        start = text_lower.find(token, pos)
                        if start == -1:
                            offset_mapping.append((0, 0))
                        else:
                            end = start + len(token)
                            offset_mapping.append((start, end))
                            pos = end

                out = {"input_ids": token_ids}
                if return_offsets_mapping:
                    out["offset_mapping"] = offset_mapping
                return out

            def decode(self, ids):
                return [self.base.inv_vocab.get(int(id), "[UNK]") for id in ids]

        self.shap_tokenizer = ShapTokenizer(tokenizer)

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
