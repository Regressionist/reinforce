1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt

Now, you can train the agents using the reinforce_experiments.ipynb notebook

To visulize hpw your models perform,
```
python -m play 
--env_name "CartPole-v1" \
--agent_path "saved_models/cartpole_reinforce_david.pth" \
--strategy "sample"
```
`--strategy` can be "sample" or "argmax". If you choose argmax, expect the rendered episode to be boring. In sample, you are sampling from the policy which will allow for more interesting episodes. If you want to record, add `--record` argument