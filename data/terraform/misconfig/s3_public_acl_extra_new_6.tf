resource "aws_s3_bucket" "unprotected_data_lake" {
  bucket = "unprotected-data-lake-bucket"
}

resource "aws_s3_bucket_acl" "unprotected_data_lake_acl" {
  bucket = aws_s3_bucket.unprotected_data_lake.id
  acl    = "public-read-write"
}