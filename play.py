import gym
import torch
import argparse
from agents import *
from torch.distributions import Categorical

def play(agent, env_name, strategy, record=False):
	env = gym.make(env_name)
	if record:
		env = gym.wrappers.RecordVideo(
			env,
			"./videos", lambda x: x % 100 == 0)
	agent.eval()
	total_reward = 0
	with torch.no_grad():
		state = env.reset()
		done = False
		while not done:
			env.render()
			state_tensor = torch.tensor(state).unsqueeze(0)
			action_probs = agent(state_tensor)[0]
			if strategy == 'argmax':
				action = torch.argmax(action_probs).item()
			else:
				m = Categorical(action_probs)
				action = m.sample().item()
			state, reward, done, info = env.step(action)
			total_reward += reward
	print(f'Total reward: {total_reward}')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some inputs')
	parser.add_argument(
		'--env_name',
		type=str,
        help='Name of the environment to run the agent on')
	parser.add_argument(
		'--agent_path',
		type=str,
        help='Location of the saved agent policy')
	parser.add_argument(
		'--strategy',
		type=str,
		choices=['argmax', 'sample'],
		default='argmax',
        help='how to choose action from probability distribution')
	parser.add_argument(
		'--record',
		default=False,
		action='store_true',
        help='if you record, no rendering')
	args = parser.parse_args()

	env_name = args.env_name
	model_pth = args.agent_path
	strategy = args.strategy
	record = args.record
	agent = torch.load(model_pth)
	play(agent, env_name, strategy, record)
