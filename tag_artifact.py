import wandb
from dotenv import load_dotenv

load_dotenv()

api      = wandb.Api()
artifact = api.artifact("adityabidappam-dayananda-sagar-university/fraud-detection/xgb-robust:latest")
artifact.aliases.append("v1")
artifact.save()
print(f"Tagged xgb-robust:latest as v1  (version: {artifact.version})")
