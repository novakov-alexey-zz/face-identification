import io.kjaer.compiletime.*
import org.emergentorder.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.backends.*
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.opencv.opencv_core.Mat
import io.bullet.borer.Cbor

import java.io.{ByteArrayOutputStream, File}
import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO
import scala.collection.parallel.CollectionConverters.*


def label2Features(dirs: Array[File]) = 
  val model = getModel
  val batchSize = 32
  val outputSize = 512

  dirs.par.map { dir =>
    val label = dir.getName
    println(s"Extractting features for '$label' at $dir folder")

    val groups = dir.listFiles.grouped(batchSize).filter(_.length == batchSize)
    val features = groups.map { files =>
      val images = files.map(f => toArray(imread(f.toString))).flatten
      val batch = files.length
      println(s"batch size: $batch, files: ${files.mkString(",")}")
      val input = Tensor(images, tensorDenotation, tensorShapeDenotation, shape(batch))           
      val out = model.fullModel[Float, "ImageClassification", "Batch" ##: "Features" ##: TSNil, 32 #: 512 #: SNil](Tuple(input))
      println(s"out size: ${out.data.length}")
      out.data.grouped(outputSize).toList
    }  
    label -> features.toList.flatten
  }

@main
def extract =
  val dirs = Paths.get("dataset-family").toFile.listFiles

  val avgFeatures = label2Features(dirs).par.map {
    (label, features) => 
      val count = features.length
      println(s"label = $label, count = $count")
      val sum = features.reduce((a, b) => a.zip(b).map(_ + _))
      label -> sum.map(_ / count)
  }.toList.toMap

  // avgFeatures.foreach { (k, v) => 
  //   println(s"label = $k, size = ${v.length}, values = ${v.sorted.mkString(",")}")
  // }

  val file = File("data/precomputed_features.cbor")
  Cbor.encode(avgFeatures).to(file).result
  

