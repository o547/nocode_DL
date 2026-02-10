from django.db import models
import uuid

# Create your models here.
class File(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False,verbose_name="ID")
    name=models.CharField(max_length=100,verbose_name="ファイル名",null=True,blank=True)
    file=models.FileField(upload_to="files/",verbose_name="csvファイル",null=True,blank=True)
    input_size=models.IntegerField(default=0,verbose_name="入力の数")
    output_size=models.IntegerField(default=0,verbose_name="出力の数")
    learned_model=models.FileField(upload_to="files/",null=True,verbose_name="学習済みモデル")
