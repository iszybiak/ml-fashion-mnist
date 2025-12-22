from train import train_model
from evaluate import evaluate_model

def run_pipeline():
    print("Start training...")
    train_model()

    print("\nStart evaluating...")
    evaluate_model()

if __name__ == '__main__':
    run_pipeline()

