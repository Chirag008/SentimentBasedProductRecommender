function loader_enable(){
    document.getElementById("loader").style.display="block";
    document.getElementsByClassName("error_message")[0].textContent="";
}

function loader_disable(){
    document.getElementById("loader").style.display="none";
}