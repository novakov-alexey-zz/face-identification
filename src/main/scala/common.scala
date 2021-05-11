import io.kjaer.compiletime.*
import org.bytedeco.javacpp.indexer.UByteIndexer
import org.emergentorder.compiletime.*
import org.bytedeco.opencv.opencv_core.Mat
import org.emergentorder.onnx.backends.*
import java.nio.file.{Files, Paths}

def shape(batch: Int) = 
  batch #: 224 #: 224 #: 3 #: SNil
  
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