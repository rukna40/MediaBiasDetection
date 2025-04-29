import torch
import torch.nn as nn
import pandas as pd
from config import QbiasConfig, MbicConfig
from data_utils import BasicTokenizer, load_data
from model import BERTModel
from train import train_epoch, evaluate
from metrics import calculate_metrics
from explain import BiasExplainer

def main():
    # config = QbiasConfig()
    config = MbicConfig()
    df = pd.read_csv(config.dataset_path)
    train_texts = df[config.text_column].dropna().tolist()
    tokenizer = BasicTokenizer()
    tokenizer.build_vocab(train_texts)
    
    train_loader, val_loader, class_weights = load_data(config, tokenizer)

    model = BERTModel(tokenizer.vocab_size).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        y_true_val, y_pred_val = evaluate(model, val_loader, config.device)
        metrics_val = calculate_metrics(
            y_true_val, y_pred_val, 
            target_names=config.bias_classes, 
            labels=list(range(len(config.bias_classes)))
        )
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {metrics_val['accuracy']:.4f}")

    torch.save(model.state_dict(), f"{config.output_dir}/qbias_model.pth")
    print(f"Model saved to {config.output_dir}/qbias_model.pth")

    model.load_state_dict(torch.load(f"{config.output_dir}/qbias_model.pth", weights_only=True))

    # Final evaluation
    y_true_test, y_pred_test = evaluate(model, val_loader, config.device)
    metrics_test = calculate_metrics(
        y_true_test, y_pred_test, 
        target_names=config.bias_classes, 
        labels=list(range(len(config.bias_classes)))
    )
    print(f"\nFinal Test Accuracy: {metrics_test['accuracy']:.4f}")
    print(metrics_test["classification_report"])

    # Print sample predictions from validation set
    sample_batch = next(iter(val_loader))
    input_ids = sample_batch['input_ids'].to(config.device)
    attention_mask = sample_batch['attention_mask'].to(config.device)
    labels = sample_batch['labels'].to(config.device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    explainer = BiasExplainer(model, tokenizer, config.device, config.bias_classes)
    N = 5
    for sample_idx in range(N):  
        text_tokens = input_ids[sample_idx].cpu().numpy()
        raw_tokens = [tokenizer.inv_vocab.get(int(idx), '[UNK]') for idx in text_tokens]
        filtered_tokens = [t for t in raw_tokens if t != '[PAD]']
        
        true_label = config.bias_classes[labels[sample_idx].item()]
        pred_label = config.bias_classes[preds[sample_idx].item()]
        
        print(f"\nSample {sample_idx+1}:")
        print("Text:", " ".join(filtered_tokens))
        print("True Label:", true_label)
        print("Predicted Label:", pred_label)
        print("Probabilities:", probs[sample_idx].cpu().numpy())

        # Generate SHAP explanation
        explanation = explainer.explain(" ".join(filtered_tokens))
        shap_values = explanation[0].values
        explanation_tokens = explanation[0].data

        print("\nToken-level SHAP contributions:")
        for token_idx, token in enumerate(explanation_tokens):
            contributions = " | ".join(
                [f"{cls}:{shap_values[token_idx, cls_idx]:.2f}" 
                for cls_idx, cls in enumerate(config.bias_classes)]
            )
            print(f"{token}: {contributions}")


    user_text = input("\nEnter your own text to analyze (or press Enter to exit): ").strip()
    while user_text:
        token_ids, attn_mask = tokenizer.encode(user_text)
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(config.device)
        mask_tensor = torch.tensor([attn_mask], dtype=torch.long).to(config.device)
        
        with torch.no_grad():
            output = model(input_tensor, mask_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
        
        # Decode tokens for display
        raw_tokens = [tokenizer.inv_vocab.get(int(idx), '[UNK]') for idx in token_ids]
        filtered_tokens = [t for t in raw_tokens if t != '[PAD]']
        
        print("\nUser Input Analysis:")
        print("Text:", " ".join(filtered_tokens))
        print("Predicted Label:", config.bias_classes[pred])
        print("Probabilities:", prob[0].cpu().numpy())
        
        # Generate SHAP explanation
        explanation = explainer.explain(user_text)
        shap_values = explanation[0].values
        explanation_tokens = explanation[0].data
        
        print("\nToken-level SHAP contributions:")
        for token_idx, token in enumerate(explanation_tokens):
            contributions = " | ".join(
                [f"{cls}:{shap_values[token_idx, cls_idx]:.2f}" 
                for cls_idx, cls in enumerate(config.bias_classes)]
            )
            print(f"{token}: {contributions}")
        
        user_text = input("\nEnter another text to analyze (or press Enter to exit): ").strip()


if __name__ == "__main__":
    main()
