import math

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def count_parameters(model):
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def grab(var):
    return var.detach().cpu().numpy()

def infinite_dataloader(dl):
    """Yield batches forever, but correctly bump the epoch on the DistributedSampler."""
    # If you’re using DistributedSampler, grab it out of the dataloader
    sampler = getattr(dl, "sampler", None)
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in dl:
            yield batch
        epoch += 1
