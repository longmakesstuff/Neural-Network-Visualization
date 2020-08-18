package de.longuyen.data

import de.longuyen.SIZE
import java.awt.Point
import java.util.*
import java.util.concurrent.ThreadLocalRandom

class CircleDataGenerator(private val size: Int) : DataGenerator {
    override fun generate(): Pair<Array<IntArray>, Array<IntArray>> {
        val xs = mutableListOf<IntArray>()
        val ys = mutableListOf<IntArray>()
        val random = Random()
        val center = Point(SIZE / 2, SIZE / 2)

        var oneCounts = 0
        while(oneCounts < size / 2) {
            val x = ThreadLocalRandom.current().nextInt(50, 150)
            val y = ThreadLocalRandom.current().nextInt(50, 150)
            val radius = center.distance(Point(x, y))
            if(radius > 100 && radius < 200){
                xs.add(intArrayOf(x, y))
                ys.add(intArrayOf(1))
                oneCounts += 1
                println(oneCounts)
            }
        }
        var zeroCounts = 0
        while(zeroCounts < size / 2){
            val x = random.nextInt(SIZE)
            val y = random.nextInt(SIZE)
            val radius = center.distance(Point(x, y))
            if(radius < 100 && radius > 200){
                xs.add(intArrayOf(x, y))
                ys.add(intArrayOf(0))
                zeroCounts += 1
                println(zeroCounts)
            }
        }

        return Pair(xs.toTypedArray(), ys.toTypedArray())
    }
}