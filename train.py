from gettext import translation
from agent.simple_setup import SimpleSetup
from agent.simple_trainer import SimpleTrainer
import argparse

def main(yaml_config_path):
    setup = SimpleSetup(yaml_config_path)
    setup.setup_pipeline()

    trainer = SimpleTrainer(setup)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config_path')
    args = parser.parse_args()
    main(args.yaml_config_path)