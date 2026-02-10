let learn_setting_button=document.querySelector(".start_learn");
learn_setting_button.addEventListener("click",()=>{
    if(document.getElementById("mid1_size").value!="" && document.getElementById("mid2_size").value!="" && document.getElementById("epoch_size").value!=""){
        document.getElementById("loading").innerHTML="<p>学習中... ブラウザの戻るボタンを押さないでください</p>";
    }
})

let input_only_csv=document.querySelector("#input_only_csv");
let input_only_csv_button=document.querySelector(".input_only_csv_button");


let check_input=()=>{
    console.log(input_only_csv.value)
    if(input_only_csv.value.indexOf(".csv")!=-1){
        input_only_csv_button.innerHTML='<button type="submit">送信</button>';
    }else{
        input_only_csv_button.innerHTML='';
    }
}


