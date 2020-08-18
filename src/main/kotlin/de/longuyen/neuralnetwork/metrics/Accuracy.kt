package de.longuyen.neuralnetwork.metrics

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable

/**
 * Accuracy for classification purpose
 */
class Accuracy : Metric(), Serializable {
    companion object {
        private const val serialVersionUID: Long = -231624563005624908
    }

    override fun compute(yTrue: INDArray, yPrediction: INDArray): Double {
        val yTrueArgMax = Nd4j.argMax(yTrue, 0).toDoubleVector()
        val yPredictionArgMax = Nd4j.argMax(yPrediction, 0).toDoubleVector()
        var ret = 0.0
        for (y in yTrueArgMax.indices) {
            if (yTrueArgMax[y] == yPredictionArgMax[y]) {
                ret += 1.0
            }
        }
        return ret / yTrueArgMax.size
    }

}