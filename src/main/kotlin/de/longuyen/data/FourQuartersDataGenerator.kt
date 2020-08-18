package de.longuyen.data

import de.longuyen.SIZE
import java.util.*

class FourQuartersDataGenerator(private val size: Int) : DataGenerator {
    override fun generate(): Pair<Array<IntArray>, Array<IntArray>> {
        val xs = mutableListOf<IntArray>()
        val ys = mutableListOf<IntArray>()
        val random = Random()
        val half = SIZE / 2
        for (i in 0 until size) {
            val x = random.nextInt(SIZE)
            val y = random.nextInt(SIZE)
            xs.add(intArrayOf(x, y))
            if (x > half) {
                if (y < half) {
                    ys.add(intArrayOf(1))
                } else {
                    ys.add(intArrayOf(0))
                }
            } else {
                if (y > half) {
                    ys.add(intArrayOf(1))
                } else {
                    ys.add(intArrayOf(0))
                }
            }
        }
        return Pair(xs.toTypedArray(), ys.toTypedArray())
    }
}