import os
import wandb
import json

def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    print("training_args", training_args)
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    # Create a simplified config dict with only serializable values
    config_dict = {}
    
    # Extract attributes from training_args
    for key, value in vars(training_args).items():
        # Skip complex objects or convert them to strings
        try:
            # Test if value is JSON serializable
            json.dumps({key: value})
            config_dict[key] = value
        except (TypeError, OverflowError):
            # For non-serializable types, convert to string representation
            config_dict[key] = str(value)

    wandb.init(
        project=training_args.wandb_project,
        entity=training_args.wandb_entity,
        config=config_dict,
        mode='online',
    )
