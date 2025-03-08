'''
CLARKE ERROR GRID ANALYSIS
The Clarke Error Grid shows the differences between a blood glucose predictive measurement and a reference measurement,
and it shows the clinical significance of the differences between these values.
The x-axis corresponds to the reference value and the y-axis corresponds to the prediction.
The diagonal line shows the prediction value is the exact same as the reference value.
This grid is split into five zones. Zone A is defined as clinical accuracy while
zones C, D, and E are considered clinical error.
Zone A: Clinically Accurate
    This zone holds the values that differ from the reference values no more than 20 percent
    or the values in the hypoglycemic range (<70 mg/dl).
    According to the literature, values in zone A are considered clinically accurate.
    These values would lead to clinically correct treatment decisions.
Zone B: Clinically Acceptable
    This zone holds values that differe more than 20 percent but would lead to
    benign or no treatment based on assumptions.
Zone C: Overcorrecting
    This zone leads to overcorrecting acceptable BG levels.
Zone D: Failure to Detect
    This zone leads to failure to detect and treat errors in BG levels.
    The actual BG levels are outside of the acceptable levels while the predictions
    lie within the acceptable range
Zone E: Erroneous treatment
    This zone leads to erroneous treatment because prediction values are opposite to
    actual BG levels, and treatment would be opposite to what is recommended.
SYNTAX:
        plot, zone = clarke_error_grid(ref_values, pred_values, title_string)
INPUT:
        ref_values          List of n reference values.
        pred_values         List of n prediciton values.
        title_string        String of the title.
OUTPUT:
        plot                The Clarke Error Grid Plot returned by the function.
                            Use this with plot.show()
        zone                List of values in each zone.
                            0=A, 1=B, 2=C, 3=D, 4=E
EXAMPLE:
        plot, zone = clarke_error_grid(ref_values, pred_values, "00897741 Linear Regression")
        plot.show()
References:
[1]     Clarke, WL. (2005). "The Original Clarke Error Grid Analysis (EGA)."
        Diabetes Technology and Therapeutics 7(5), pp. 776-779.
[2]     Maran, A. et al. (2002). "Continuous Subcutaneous Glucose Monitoring in Diabetic
        Patients" Diabetes Care, 25(2).
[3]     Kovatchev, B.P. et al. (2004). "Evaluating the Accuracy of Continuous Glucose-
        Monitoring Sensors" Diabetes Care, 27(8).
[4]     Guevara, E. and Gonzalez, F. J. (2008). Prediction of Glucose Concentration by
        Impedance Phase Measurements, in MEDICAL PHYSICS: Tenth Mexican
        Symposium on Medical Physics, Mexico City, Mexico, vol. 1032, pp.
        259261.
[5]     Guevara, E. and Gonzalez, F. J. (2010). Joint optical-electrical technique for
        noninvasive glucose monitoring, REVISTA MEXICANA DE FISICA, vol. 56,
        no. 5, pp. 430434.
'''

import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
# End Custom libraries

# This function takes in the reference values and the prediction values as lists and returns a list with each index corresponding to the total number
# of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot
def clarke_error_grid(ref_values, pred_values, title_string):
    # Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(
        len(ref_values), len(pred_values))

    # Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if ref_values.max() > 400 or pred_values.max() > 400:
        print(
            "Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(
                max(ref_values), max(pred_values)))
    if ref_values.min() < 0 or pred_values.min() < 0:
        print(
            "Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(
                min(ref_values), min(pred_values)))

    # Clear plot
    plt.clf()

    # Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='orange', s=1)
    plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

    # Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400) / (400))

    # Plot zone lines
    plt.plot([0, 400], [0, 400], ':', c='black')  # Theoretical 45 regression line
    plt.plot([0, 175 / 3], [70, 70], '-', c='black')
    # plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175 / 3, 400 / 1.2], [70, 400], '-',
             c='black')  # Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400], '-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290], [180, 400], '-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')  # Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320], '-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180], '-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    # Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

    # Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values.iloc[i] <= 70 and pred_values.iloc[i] <= 70) or (
                pred_values.iloc[i] <= 1.2 * ref_values.iloc[i] and pred_values.iloc[i] >= 0.8 * ref_values.iloc[i]):
            zone[0] += 1  # Zone A

        elif (ref_values.iloc[i] >= 180 and pred_values.iloc[i] <= 70) or (
                ref_values.iloc[i] <= 70 and pred_values.iloc[i] >= 180):
            zone[4] += 1  # Zone E

        elif ((ref_values.iloc[i] >= 70 and ref_values.iloc[i] <= 290) and pred_values.iloc[i] >= ref_values.iloc[
            i] + 110) or ((ref_values.iloc[i] >= 130 and ref_values.iloc[i] <= 180) and (
                pred_values.iloc[i] <= (7 / 5) * ref_values.iloc[i] - 182)):
            zone[2] += 1  # Zone C
        elif (ref_values.iloc[i] >= 240 and (pred_values.iloc[i] >= 70 and pred_values.iloc[i] <= 180)) or (
                ref_values.iloc[i] <= 175 / 3 and pred_values.iloc[i] <= 180 and pred_values.iloc[i] >= 70) or (
                (ref_values.iloc[i] >= 175 / 3 and ref_values.iloc[i] <= 70) and pred_values.iloc[i] >= (6 / 5) *
                ref_values.iloc[i]):
            zone[3] += 1  # Zone D
        else:
            zone[1] += 1  # Zone B

    return plt, zone


def zone_percentages(file, zone):
    total_points = zone[0] + zone[1] + zone[2] + zone[3] + zone[4]
    A_zone = float(zone[0]/total_points)*100
    B_zone = float(zone[1] / total_points) * 100
    C_zone = float(zone[2] / total_points) * 100
    D_zone = float(zone[3] / total_points) * 100
    E_zone = float(zone[4] / total_points) * 100
    percentages = [(file, A_zone, B_zone, C_zone, D_zone, E_zone)]
    total_percentages = pd.DataFrame(percentages, columns=['Model', 'A_zone', 'B_zone', 'C_zone', 'D_zone', 'E_zone'])
    return total_percentages


def main():
    model_name = 't_12_q_5_l_2_N1_64_N2_64_lr_0.001_nr_128_lr2_1e-05_True_notree'
    all_percentages = pd.DataFrame(columns=['Model', 'A_zone', 'B_zone', 'C_zone', 'D_zone', 'E_zone'])
    all_values = pd.DataFrame(columns=['Y_1', 'y_hat_1', 'clave'])
    for i in [559, 563, 570, 575, 588, 591]:
        data_dir = 'dataset_low_pass_filter/test/{}/'.format(i)
        models_root = 'weights/{}/{}'.format(i, model_name)

        parameters = model_name.split('_')
        model_type = 'notree'
        feat = 4
        t_seq = int(parameters[1])
        q = int(parameters[3])
        H = 1
        num_layers = int(parameters[5])
        layer1 = int(parameters[7])
        layer2 = int(parameters[9])
        lr = (parameters[11])
        dropout, recurrent_dropout = 0, 0

        with open(models_root + '/model_config.json') as json_file:
            json_config = json_file.read()
        model = tf.keras.models.model_from_json(json_config)
        weights_dir = models_root + '/model.tf'
        model.load_weights(weights_dir).expect_partial()

        original_dataset = pd.read_csv(data_dir + 'Patient_{}_testing.xlsx'.format(i))
        original_dataset = original_dataset[144 - t_seq:]
        testX_OG = []
        testY_OG = []
        testY_OG_PH = []
        for i in range(len(original_dataset) - t_seq - q - H):
            if original_dataset[i + t_seq + q + H:i + t_seq + q + H + 1]['old_CGM'].values != 0:
                testX_OG.append(original_dataset[i:i + t_seq][['old_CGM', 'total_insulin', 'meal', 'HR']])
                testY_OG.append(original_dataset[i + t_seq + q + H:i + t_seq + q + H + 1][['old_CGM', 'clave']])
            if original_dataset[i + t_seq:i + t_seq + 1]['old_CGM'].values != 0:
                testY_OG_PH.append(original_dataset[i + t_seq:i + t_seq + 1][['old_CGM', 'clave']])
        testX = np.array(testX_OG)
        testY = np.array(testY_OG)

        model.compile(loss=tf.keras.metrics.RootMeanSquaredError(),
                      optimizer='adam',
                      metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.metrics.MeanAbsoluteError()])

        Y = testY[:, :, 0].astype('float32')
        Y = Y.flatten()
        testX = testX.astype('float32')
        if model_type == 'tree' and feat >= 4:
            eval = model.evaluate({"Base_model": testX[:, :, 0:3], "HR": testX[:, :, 3:feat]}, Y)
            y_hat = model.predict({"Base_model": testX[:, :, 0:3], "HR": testX[:, :, 3:feat]})
        else:
            eval = model.evaluate(testX, Y)
            y_hat = model.predict(testX)

        testY = testY.reshape([len(testY), 2])
        testY = pd.DataFrame(testY, columns=['Y_1', 'clave'])
        testY['y_hat_1'] = y_hat
        all_values = pd.concat([all_values, testY])

    #plot, zone = clarke_error_grid(all_values['Y_1'], all_values['y_hat_1'], 'tseq_{}_H_1_q_{}'.format(t_seq, q))
    plot, zone = clarke_error_grid(all_values['Y_1'], all_values['y_hat_1'], 'Substitution Model')
    plot.savefig('results/{}'.format(model_name), format='eps')
    total_percentages = zone_percentages(model_name, zone)
    all_percentages = pd.concat([all_percentages, total_percentages])
    all_percentages.to_csv('results/{}_{}'.format(feat, model_type), header=True, index=False, sep='\t')

if __name__ == '__main__':
    main()
