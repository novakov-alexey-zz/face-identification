import io.kjaer.compiletime.*
import org.bytedeco.javacpp.indexer.UByteIndexer
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.opencv.opencv_core.Mat
import org.emergentorder.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.backends.*
import io.bullet.borer.Cbor

import java.io.{ByteArrayOutputStream, File}
import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO
import scala.collection.parallel.CollectionConverters.*

val shape =                    1     #:     224      #:    224    #: 3     #: SNil
val tensorShapeDenotation = "Batch" ##: "Height" ##: "Width" ##: "Channel" ##: TSNil
val tensorDenotation: String & Singleton = "Image"

def toArray(mat: Mat): Array[Float] =
  val w = mat.cols
  val h = mat.rows
  val c = mat.channels
  val rowStride = w * c

  val result = new Array[Int](rowStride * h)
  val indexer = mat.createIndexer[UByteIndexer]()
  var off = 0
  var y = 0
  while (y < h) 
    indexer.get(y, result, off, rowStride)
    off += rowStride
    y += 1
    
  result.map(_.toFloat)

def getModel = 
  val bytes = Files.readAllBytes(Paths.get("data", "model.onnx"))
  ORTModelBackend(bytes)

lazy val model = getModel

def label2Features(dirs: Array[File]) = 
  dirs.par.map { dir =>
    val label = dir.getName
    println(s"Extractting features for '$label' at $dir folder")
    val features = for f <- dir.listFiles yield
      println(f)
      val image = toArray(imread(f.toString))
      val input = Tensor(image, tensorDenotation, tensorShapeDenotation, shape)
      val out = model.fullModel[Float, "ImageNetClassification", "Batch" ##: "Features" ##: TSNil, 1 #: 512 #: SNil](Tuple(input))
      out.data
    label -> features
  }

@main
def extract =
  val dirs = Paths.get("dataset-family").toFile.listFiles

  val avgFeatures = label2Features(dirs).par.map {
    (label, features) => 
      val count = features.length
      val sum = features.reduce((a, b) => a.zip(b).map(_ + _))
      label -> sum.map(_ / count)
  }.toList.toMap

  avgFeatures.foreach { (k, v) => 
    println(s"$k: ${v.sorted.mkString(",")}")
  }

  val file = File("data/precomputed_features.cbor")
  Cbor.encode(avgFeatures).to(file).result
  

