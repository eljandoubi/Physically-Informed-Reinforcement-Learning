import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import wandb  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from simple_parsing import ArgumentParser  # noqa: E402
from tqdm import trange  # noqa: E402

from pirl.ppo import PPO, TrainingConfig  # noqa: E402
from pirl.robotenv import SubprocVecEnv, get_env_infos, make_env  # noqa: E402
from pirl.rollout import RolloutBuffer  # noqa: E402
from pirl.video import video_render  # noqa: E402


def main(config: TrainingConfig = TrainingConfig()):
    exit_code = 0
    env = None
    run = None
    try:
        if config.use_wandb:
            load_dotenv()
            run = wandb.init(
                project="ppo-robosuite",
                config=config.__dict__,
                id=config.runid,
                resume="allow",
            )
            config.set_id(run.id)
        config.update_path()
        if run is not None:
            wandb.config.update(
                {"checkpoint_dir": config.checkpoint_dir, "runid": run.id},
                allow_val_change=True,
            )
        checkpoint_best_path = f"{config.checkpoint_dir}/PPO_{config.env_name}_best.pth"
        episode_log_path = Path(config.checkpoint_dir) / "episodes.jsonl"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        keys = [
            "robot0_eye_in_hand_image",
            "robot0_eye_in_hand_depth",
            "robot0_proprio-state",
        ]
        if config.use_object_obs:
            keys.append("object-state")
        action_dim, obs_shapes = get_env_infos(
            config.img_size, keys, use_object_obs=config.use_object_obs
        )

        print("MUJOCO_GL:", os.getenv("MUJOCO_GL"))
        env_kwargs = dict(
            device_id=device.index if os.getenv("MUJOCO_GL") == "egl" else -1,
            img_size=config.img_size,
            max_episode_steps=config.max_ep_len,
            reward_shaping=config.reward_shaping,
            use_object_obs=config.use_object_obs,
        )
        print("Creating vectorized environment with the following kwargs:", env_kwargs)
        env = SubprocVecEnv(
            make_env,
            num_envs=config.num_envs,
            env_kwargs=env_kwargs,
            filter_keys=keys,
        )

        buffer = RolloutBuffer(
            config.max_ep_len, config.num_envs, obs_shapes, action_dim, device
        )
        print(buffer)
        ppo_agent = PPO(action_dim, device, obs_shapes, config)

        if config.load_checkpoint_path:
            ppo_agent.load(config.load_checkpoint_path)

        time_step = 0

        num_episodes = int(config.max_training_timesteps // config.update_timestep) + 1
        best_reward = float("-inf")
        with trange(num_episodes, desc="Episodes") as pbar:
            for ep in range(num_episodes):
                obs = env.reset()
                current_ep_reward = 0
                for t in range(1, config.max_ep_len + 1):
                    # select action with policy
                    action, log_prob, values, obs = ppo_agent.select_action(obs)
                    env_actions = action.detach().cpu().numpy().clip(-1, 1)
                    next_obs, rewards, dones = env.step(env_actions)
                    buffer.add(obs, action, rewards, dones, values, log_prob)
                    obs = next_obs
                    time_step += 1
                    current_ep_reward += torch.mean(rewards).item()
                    completed = [
                        {
                            "step": time_step * config.num_envs,
                            "env_id": index,
                            "success": success,
                        }
                        for index, success in enumerate(env.completed_successes)
                        if success is not None
                    ]
                    if completed:
                        with episode_log_path.open("a", encoding="utf-8") as episode_log:
                            for episode in completed:
                                episode_log.write(json.dumps(episode) + "\n")

                if current_ep_reward > best_reward:
                    best_reward = current_ep_reward
                    print(f"New best reward: {best_reward:.2f} at episode {ep}")
                    # save model checkpoint
                    ppo_agent.save(checkpoint_best_path)

                # update PPO agent
                if len(buffer) >= config.update_timestep:
                    assert len(buffer) == config.update_timestep, (
                        f"Buffer length {len(buffer)} does not match expected "
                        f"{config.update_timestep}"
                    )
                    ppo_agent.update(buffer, next_obs)
                    buffer.reset()

                    # save model checkpoint
                    checkpoint_path = (
                        f"{config.checkpoint_dir}/PPO_{config.env_name}_{time_step}.pth"
                    )
                    ppo_agent.save(checkpoint_path)

                pbar.update(1)
                pbar.set_postfix({"Timestep": time_step, "Reward": current_ep_reward})

                # Log to wandb
                log_payload = {
                    "reward": current_ep_reward,
                }
                if ppo_agent.mse_losses:
                    log_payload["mse_loss"] = ppo_agent.mse_losses[-1]
                if ppo_agent.entropy_losses:
                    log_payload["entropy_loss"] = ppo_agent.entropy_losses[-1]
                if ppo_agent.surrogate_losses:
                    log_payload["surrogate_loss"] = ppo_agent.surrogate_losses[-1]
                if ppo_agent.obj_pred_losses:
                    log_payload["obj_pred_loss"] = ppo_agent.obj_pred_losses[-1]

                if run is not None:
                    wandb.log(log_payload, step=time_step)

        video_render(config, agent=ppo_agent)
    except BaseException:
        exit_code = 1
        raise

    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if run is not None:
                wandb.finish(exit_code=exit_code)


def cli():
    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig, dest="training_config")
    args = parser.parse_args()
    config = getattr(args, "training_config")
    main(config)


if __name__ == "__main__":
    cli()
