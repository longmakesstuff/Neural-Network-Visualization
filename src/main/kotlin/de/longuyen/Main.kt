package de.longuyen

import de.longuyen.data.CircleDataGenerator
import de.longuyen.gui.Frame

fun main(){
    val data = CircleDataGenerator(2000).generate()
    val frame = Frame(data.first, data.second)
    frame.run()
}