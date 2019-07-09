import yaml

class Config():
    def __init__(self,config):
        data = self.config_read(config)
        for key,value in data.items():
            setattr(self,key,value)

    def config_read(self,config):
        ycon={}
        with open(config, 'r') as stream:
            try:
                ycon = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return ycon
