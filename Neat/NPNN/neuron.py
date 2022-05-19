class Neuron:
    def __init__(self, output_length, neuron_type=0, action_index=0, sensory_index=0):
        """
        Types:

        2: Action
        1: Sensory
        0: Normal
        """
        self.neuron_type = int(neuron_type)
        self.action_index = int(action_index)
        self.sensory_index = int(sensory_index)
        self.output_length = output_length

        self.energy = 0

        self.h_id = None

    def step(self, input_array):
        out = [0 for i in range(self.output_length)]

        if self.neuron_type == 2:
            out[self.action_index] = self.energy
            self.energy = 0

        elif self.neuron_type == 1:
            self.energy = input_array[self.sensory_index]

        else:
            self.energy = 0

        return out
