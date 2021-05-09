import java.nio.file.{Files, Paths}
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.backends._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._

import java.awt.event.KeyEvent
import javax.swing.{JFrame, WindowConstants}
import java.nio.ByteBuffer
import org.bytedeco.opencv.opencv_core.{Mat, Size, UMat, RectVector, Rect, Point, Scalar}
import org.bytedeco.opencv.opencv_imgproc.*
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.opencv.global.opencv_highgui.*
import org.bytedeco.opencv.opencv_videoio.*
import org.bytedeco.opencv.global.opencv_videoio.*
import org.bytedeco.javacpp.indexer.UByteIndexer
import org.bytedeco.javacv.{CanvasFrame, OpenCVFrameConverter}
import scala.collection.mutable.ArrayBuffer
import org.bytedeco.opencv.opencv_objdetect.*

@main
def demo() =            
  val faceCascade = CascadeClassifier("cv2_cascades/haarcascades/haarcascade_frontalface_alt2.xml")
  val capture = VideoCapture(0)
  // capture.set(CAP_PROP_FRAME_WIDTH, 1280)
  // capture.set(CAP_PROP_FRAME_HEIGHT, 720)
  val canvasFrame = CanvasFrame("Detected Faces")
  canvasFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
  canvasFrame.setCanvasSize(1280, 720)
  val frame = Mat()
  val converter = OpenCVFrameConverter.ToMat()

  try
    while (capture.read(frame))
      val gray = UMat()
      frame.copyTo(gray)
      cvtColor(gray, gray, COLOR_BGR2GRAY)
      equalizeHist(gray, gray)
      val faces = RectVector()
      faceCascade.detectMultiScale(gray, faces)
      println(s"Faces detected: {faces.size()}")  
      
      for f <- faces.get do
        rectangle(frame,
          Point(f.x, f.y),
          Point(f.x + f.width, f.y + f.height),
          Scalar(0, 255, 0, 1)
        )
        val rectCrop = Rect(f.x, f.y, f.width, f.height)
        val cropped = Mat(frame, rectCrop)
        //println(toIntArray(cropped).mkString(","))
              
      canvasFrame.showImage(converter.convert(frame))                              
  finally
    capture.release()          
    canvasFrame.dispose()  
  val image = imread("test_family_images/test3.jpg")    
  // resize(image, image, new Size(224,224))
  // val data = ??? //toIntArray(image).map(_.toFloat)

  // val (model, input) = getModel(data)

  // val out = model.fullModel[Float,
  //                            "ImageNetClassification",
  //                            "Batch" ##: "Features" ##: TSNil,
  //                            1 #: 512 #: SNil](Tuple(input))

  // println(out.data.mkString(","))