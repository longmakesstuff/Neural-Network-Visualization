package de.longuyen.neuralnetwork.optimizers

import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import java.io.Serializable

/**
 * AdaGrad (for adaptive gradient algorithm) is a modified stochastic gradient descent algorithm with per-parameter learning rate.
 */
class AdaGrad(private val learningRate: Double, private val eps: Double=0.0001): Optimizer(), Serializable {
    companion object {
        private const val serialVersionUID: Long = 1
    }

    private val cache = mutableMapOf<String, INDArray>()
    private var initialized = false
    override fun optimize(weights: MutableMap<String, INDArray>, gradients: MutableMap<String, INDArray>, layers: Int) {
        if(!initialized){
            for(i in 2 until layers){
                cache["cW$i"] = Nd4j.zerosLike(gradients["dW$i"]!!).castTo(DataType.DOUBLE)
                cache["cb$i"] = Nd4j.zerosLike(gradients["db$i"]!!).castTo(DataType.DOUBLE)
            }
            initialized = true
        }
        for(i in 2 until layers){
            cache["cW$i"] = cache["cW$i"]!!.add(Transforms.pow(gradients["dW$i"]!!, 2.0))
            weights["W$i"] = weights["W$i"]!!.sub(((Transforms.sqrt(cache["cW$i"]).mul(learningRate)).add(eps)).mul(gradients["dW$i"]!!))

            cache["cb$i"] = cache["cb$i"]!!.add(Transforms.pow(gradients["db$i"]!!, 2.0))
            weights["b$i"] = weights["b$i"]!!.sub(((Transforms.sqrt(cache["cb$i"]).mul(learningRate)).add(eps)).mul(gradients["db$i"]!!))
        }
    }

}