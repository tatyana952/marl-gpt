import argparse
import torch


def merge_checkpoints(gpt_path: str, ppo_path: str, output_path: str) -> None:
    checkpoint_gpt = torch.load(gpt_path)

    checkpoint_model = torch.load(ppo_path, weights_only=False)
    state_dict = checkpoint_model["model"]
    model1_state_dict = {
        k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")
    }

    checkpoint_gpt["model"] = model1_state_dict
    torch.save(checkpoint_gpt, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge PPO model weights into a GPT checkpoint."
    )
    parser.add_argument(
        "--gpt-path",
        type=str,
        default="out/ckpt.pt",
        help="Path to the GPT checkpoint (.pt).",
    )
    parser.add_argument(
        "--ppo-path",
        type=str,
        default="train_dir/best.pth",
        help="Path to the PPO model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="out/ppo_model.pt",
        help="Path to save the merged checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_checkpoints(args.gpt_path, args.ppo_path, args.output_path)


if __name__ == "__main__":
    main()
