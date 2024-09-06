import math


class HyperParameterSchedule:
    def get(self, step: int) -> float:
        raise NotImplementedError


class ConstantHyperParameterSchedule(HyperParameterSchedule):
    def __init__(self, value: float):
        self.value = value

    def get(self, _step: int) -> float:
        return self.value
    
class LinearHyperParameterSchedule(HyperParameterSchedule):
    def __init__(self, initial_value: float, final_value: float, decay_rate: float):
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_rate = decay_rate

    def get(self, step: int) -> float:
        if step >= self.decay_rate:
            return self.final_value
        return self.initial_value + step * (self.final_value - self.initial_value) / self.decay_rate


class ExponentialHyperParameterSchedule(HyperParameterSchedule):
    def __init__(self, initial_value: float, final_value: float, decay_rate: float):
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_rate = decay_rate

    def get(self, step: int) -> float:
        return self.final_value + (self.initial_value - self.final_value) * math.exp(
            -1.0 * step / self.decay_rate
        )
