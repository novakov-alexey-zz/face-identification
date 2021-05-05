import ammonite.ops._

val wd = pwd / 'family

val ratio = 0.2 // split ratio
val targetDir = pwd / "dataset-family"
rm ! targetDir
mkdir ! targetDir

val members = ls ! wd
members.foreach { member =>
  val files = ls ! member
  val testRatio = files.length * ratio
  val (test, train) = files.splitAt(testRatio.toInt)
  copyFiles("test", test, member.last)
  copyFiles("train", train, member.last)
}

def copyFiles(dirName: String, files: Seq[Path], label: String) = {
  val testDir = targetDir / dirName / label
  mkdir ! testDir
  files.foreach { f =>
    println("Copy " + f)
    cp(f, testDir / f.last)
  }
}
