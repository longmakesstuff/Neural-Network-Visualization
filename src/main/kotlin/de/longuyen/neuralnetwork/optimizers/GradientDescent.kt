package de.longuyen.neuralnetwork.optimizers

import org.nd4j.linalg.api.ndarray.INDArray
import java.io.Serializable

class GradientDescent (private val learningRate: Double) : Optimizer(), Serializable {
    companion object {
        private const val serialVersionUID: Long = 1
    }
    override fun optimize(weights: MutableMap<String, INDArray>, gradients: MutableMap<String, INDArray>, layers: Int) {
        for (i in 2 until layers) {
            weights["W$i"] = weights["W$i"]!!.add(gradients["dW$i"]!!.mul(learningRate))
            weights["b$i"] = weights["b$i"]!!.add(gradients["db$i"]!!.mul(learningRate))
        }
    }

}