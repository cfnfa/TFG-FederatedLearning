# Configuracion optimizacion de parametros
class CombinacionParametrosRedOptimizar():
    def __init__(self):
        self.combinaciones_t_seq = [12]  # Input dimension a la red
        self.combinaciones_q = [5]  # Prediction horizon is H+q
        self.combinaciones_n_layers = [2]
        self.neurons_layer_1 = [64]
        self.neurons_layer_2 = [64]
        self.neurons_layer_3 = [0]
        self.fixed_neurons_1 = 16
        self.fixed_neurons_2 = 0
        self.fixed_neurons_3 = 4
        self.combinaciones_lr = [1e-3]  # learning rate
        self.fixed_lr = 1e-3
        self.dropout = 0.8
        self.recurrent_dropout = 0.0

    def get_combinaciones_tseq(self):
        return self.combinaciones_t_seq

    def get_dropout(self):
        return self.dropout, self.recurrent_dropout

    def get_combinaciones_q(self):
        return self.combinaciones_q

    def get_combinaciones_n_layers(self):
        return self.combinaciones_n_layers

    def get_combinaciones_optimizer_alg(self):
        return self.combinaciones_optimizer_alg

    def get_neuronas_1(self):
        return self.neurons_layer_1[0], self.neurons_layer_1[1], self.neurons_layer_1[2]

    def get_neuronas_2(self):
        return self.neurons_layer_2[0], self.neurons_layer_2[1], self.neurons_layer_2[2]

    def get_neuronas_3(self):
        return self.neurons_layer_3[0], self.neurons_layer_3[1], self.neurons_layer_3[2]

    def get_fixed_neurons(self):
        return self.neurons_layer_1, self.neurons_layer_2

    def get_combinaciones_name_optimizer(self):
        return self.combinaciones_name_optimizer

    def get_combinaciones_lr(self):
        return self.combinaciones_lr

    def get_fixed_lr(self):
        return self.fixed_lr

    def n_total_comb_opt_params_neuronasIguales_layers(self):
        n_combinaciones = 0
        for t_seq in self.combinaciones_t_seq:
            for q in self.combinaciones_q:  # Prediction horizon is H+q
                for n_layers in self.combinaciones_n_layers:
                    for n_neuronas in self.combinacion_neuronas_iguales_en_cada_layer:
                        n_combinaciones += 1

        return n_combinaciones

    def n_total_comb_opt_params_neuronasDistintas_layers(self):
        n_combinaciones = 0
        for t_seq in self.combinaciones_t_seq:
            for q in self.combinaciones_q:  # Prediction horizon is H+q
                for n_layers in self.combinaciones_optimizer_alg:
                    for n_neuronas in self.combinacion_neuronas_iguales_en_cada_layer:
                        if n_layers >= 2:
                            for n_neuronas2 in self.combinacion_neuronas_iguales_en_cada_layer:
                                if n_layers >= 3:
                                    for n_neuronas3 in self.combinacion_neuronas_iguales_en_cada_layer:
                                        n_combinaciones += 1

                                else:
                                    n_neuronas3 = 0
                                    n_combinaciones += 1

                        else:
                            n_neuronas2 = 0
                            n_neuronas3 = 0
                            n_combinaciones += 1

        return n_combinaciones

    def n_total_comb_opt_params_lr_optimizer(self):
        n_combinaciones = 0
        for t_seq in self.combinaciones_t_seq:
            for q in self.combinaciones_q:  # Prediction horizon is H+q
                for n_layers in self.combinaciones_optimizer_alg:
                    for n_neuronas in self.combinaciones_lr:
                        n_combinaciones += 1

        return n_combinaciones

    def n_total_comb_redes_def(self):
        n_combinaciones = 0
        for t_seq in self.combinaciones_t_seq:
            for q in self.combinaciones_q:  # Prediction horizon is H+q
                n_combinaciones += 1

        return n_combinaciones


if __name__ == '__main__':
    combi_redes = CombinacionParametrosRedOptimizar()
    combinacion_neuronas_fijas = combi_redes.get_fixed_neurons()
    combinacion_n_layers = combi_redes.get_combinaciones_n_layers()
    combinaciones_t_seq = combi_redes.get_combinaciones_tseq()
    combinaciones_q = combi_redes.get_combinaciones_q()
    print("Combinacion de neuronas fijas: ", combinacion_neuronas_fijas)
    print("Combinacion de Layers: ", combinacion_n_layers)

    print("Numero de combinaciones Optimizacion parametros 01 Neuronas Layers: ",
          combi_redes.n_total_comb_opt_params_neuronasIguales_layers())
    print("Numero de combinaciones Optimizacion parametros 01 Neuronas distintas Layers: ",
          combi_redes.n_total_comb_opt_params_neuronasDistintas_layers())
    print("Numero de combinaciones Optimizacion parametros 02 LR Optimizer: ",
          combi_redes.n_total_comb_opt_params_lr_optimizer())
    print("Numero de combinaciones Redes definitivas: ", combi_redes.n_total_comb_redes_def())
