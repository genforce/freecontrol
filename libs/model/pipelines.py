pipelines = {}
def register_pipeline(name):
    def decorator(cls):
        pipelines[name] = cls
        return cls
    return decorator

def make_pipeline(name, model_id, **kwargs):
    return pipelines[name].from_pretrained(model_id,**kwargs)
