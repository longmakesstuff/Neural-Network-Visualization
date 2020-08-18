package de.longuyen.neuralnetwork.metrics

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import java.io.Serializable

/**
 * Mean absolute error for regression metric
 */
class MAE : Metric(), Serializable {
    companion object {
        private const val serialVersionUID: Long = 1
    }

    override fun compute(yTrue: INDArray, yPrediction: INDArray): Double {
        val diff = yPrediction.sub(yTrue)
        val absolute = Transforms.abs(diff)
        return absolute.sum(true, 0).mean().element() as Double
    }

}