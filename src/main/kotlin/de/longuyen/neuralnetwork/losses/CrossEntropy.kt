package de.longuyen.neuralnetwork.losses

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import java.io.Serializable

/**
 * Cross entropy loss for classification purpse
 */
class CrossEntropy : LossFunction(), Serializable {
    companion object {
        private const val serialVersionUID: Long = 3107302032578286059
    }

    override fun forward(yTrue: INDArray, yPrediction: INDArray): INDArray {
        val loss = (yTrue.mul(Transforms.log(yPrediction))).sum(true, 0)
        val negativeLoss = loss.mul(-1)
        val cost = negativeLoss.sum(1)
        val mean = cost.div(yTrue.shape()[1])
        return mean
    }

    override fun backward(yTrue: INDArray, yPrediction: INDArray): INDArray {
        return ((yTrue.sub(yPrediction)).div(yTrue.shape()[1])).mul(-1.0)
    }
}