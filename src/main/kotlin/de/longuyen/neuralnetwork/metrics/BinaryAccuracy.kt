package de.longuyen.neuralnetwork.metrics

import org.nd4j.linalg.api.ndarray.INDArray
import java.io.Serializable


/**
 * Accuracy for classification purpose
 */
class BinaryAccuracy : Metric(), Serializable {
    companion object {
        private const val serialVersionUID: Long = -231624563005624908
    }

    override fun compute(yTrue: INDArray, yPrediction: INDArray): Double {
        val yTrueVector = yTrue.toDoubleVector()
        val yPredictionVector = yPrediction.toDoubleVector()
        var ret = 0.0
        for (y in yTrueVector.indices) {
            val rounded = if(yPredictionVector[y] > 0.5){
                1.0
            }else{
                0.0
            }
            if (rounded == yTrueVector[y]) {
                ret += 1.0
            }
        }
        return ret / yPredictionVector.size
    }
}