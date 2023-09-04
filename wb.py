import wandb


def init_wandb(api_key: str, project: str, config: dict) -> wandb:
    wandb.login(key=api_key)
    wandb.init(project=project, config=config)
    return wandb