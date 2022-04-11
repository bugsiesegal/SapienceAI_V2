from neat.attributes import BaseAttribute
from random import choice, gauss, random, uniform

class IntAttribute(BaseAttribute):
    _config_items = {
        "init_mean": [int, None],
        "init_stdev": [int, None],
        "init_type": [str, 'gaussian'],
        "replace_rate": [float, None],
        "mutate_rate": [float, None],
        "mutate_power": [int, None],
        "max_value": [int, None],
        "min_value": [int, None]
    }

    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)

    def init_value(self, config):
        mean = getattr(config, self.init_mean_name)
        stdev = getattr(config, self.init_stdev_name)
        init_type = getattr(config, self.init_type_name).lower()

        if ('gauss' in init_type) or ('normal' in init_type):
            return int(self.clamp(gauss(mean, stdev), config))

        if 'uniform' in init_type:
            min_value = max(getattr(config, self.min_value_name),
                            (mean - (2 * stdev)))
            max_value = min(getattr(config, self.max_value_name),
                            (mean + (2 * stdev)))
            return int(uniform(min_value, max_value))

        raise RuntimeError("Unknown init_type {!r} for {!s}".format(getattr(config,
                                                                            self.init_type_name),
                                                                    self.init_type_name))

    def mutate_value(self, value, config):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return int(self.clamp(value + gauss(0.0, mutate_power), config))

        replace_rate = getattr(config, self.replace_rate_name)

        if r < replace_rate + mutate_rate:
            return int(self.init_value(config))

        return int(value)

    def validate(self, config):  # pragma: no cover
        pass