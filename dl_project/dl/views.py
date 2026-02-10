from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from .models import *
from .func import learn
from .model_culculate import culculate
from .csv_output import csv_out

# Create your views here.


class IndexView(View):
    def get(self, request):
        files = File.objects.all()
        return render(request, "dl/index.html", {"files": files})


class AddView(View):
    def get(self, request):
        return render(request, "dl/add.html")

    def post(self, request):
        file = request.FILES.get("file")
        input_size = request.POST.get("input_size")
        output_size = request.POST.get("output_size")
        if str(file).rfind(".csv") != -1:
            File.objects.create(
                name=str(file).replace(".csv", ""),
                file=file,
                input_size=input_size,
                output_size=output_size,
            )
        return redirect("dl:index")


class FileView(View):
    def get(self, request, request_uuid):
        file = get_object_or_404(File, id=request_uuid)
        print(str(file))
        outputs = []
        inputs = [""] * file.input_size
        return render(
            request,
            "dl/file.html",
            {"file": file, "inputs": inputs, "outputs": outputs},
        )

    def post(self, request, request_uuid):
        file = get_object_or_404(File, id=request_uuid)
        inputs = [""] * file.input_size
        outputs = []
        output_csv_name = None
        if request.POST.get("epoch_size") is not None:
            # モデルの学習
            mid1_size = request.POST.get("mid1_size")
            mid2_size = request.POST.get("mid2_size")
            epoch_size = request.POST.get("epoch_size")
            learn(
                file_path="static/dl/files/" + file.name + ".csv",
                file_name=file.name,
                epoch_size=int(epoch_size),
                input_size=file.input_size,
                mid1_size=int(mid1_size),
                mid2_size=int(mid2_size),
                output_size=file.output_size,
            )
            file.learned_model = "static/dl/files/" + file.name + ".pth"
            file.save()
        elif request.POST.get("input0") is not None:
            # 手入力から予想を出力
            inputs = []
            for i in range(file.input_size):
                inputs.append(float(request.POST.get("input" + str(i))))
            outputs = culculate(
                file_path="static/dl/files/" + file.name + ".pth", inputs=inputs
            )
            outputs = [round(x, 3) for x in outputs]
        else:
            input_only_csv = request.FILES.get("input_only_csv")
            fs = FileSystemStorage(location="static/dl/files")
            saved_name = fs.save(input_only_csv.name, input_only_csv)
            csv_out(
                file_path="static/dl/files/" + saved_name,
                model_path="static/dl/files/" + file.name + ".pth",
                input_size=file.input_size,
                output_size=file.output_size,
            )
            output_csv_name = saved_name.replace(".csv", "_out.csv")

        return render(
            request,
            "dl/file.html",
            {
                "file": file,
                "inputs": inputs,
                "outputs": outputs,
                "output_csv_name": output_csv_name,
            },
        )


class EditView(View):
    def get(self, request, request_uuid):
        print("edit実行")
        file = get_object_or_404(File, id=request_uuid)
        file.delete()
        return redirect("dl:index",)


index = IndexView.as_view()
add = AddView.as_view()
file = FileView.as_view()
edit = EditView.as_view()
