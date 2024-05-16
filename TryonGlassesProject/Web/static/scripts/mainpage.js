socket = io();
socket.connect('http://127.0.0.1:5000/');

socket.on('connect', function(){

})

function Chup() {
    socket.emit('chup')
}

function DeXuat() {
    var list, index;
    list = document.getElementsByClassName("list-group-item");
    for (index = 0; index < list.length; ++index) {
        list[index].setAttribute('style', 'display:none !important');
    }
    socket.emit('deXuat')
}

function TatCa() {
    var list, index;
    list = document.getElementsByClassName("list-group-item");
    for (index = 0; index < list.length; ++index) {
        list[index].setAttribute('style', 'display:block !important');
    }
}

function ChonKieng(ctrl) {
    var IDKieng = ctrl.getElementsByTagName('img')[0].getAttribute('src');
    var LinkKieng = ctrl.getElementsByTagName('a')[0].getAttribute('href');
    socket.emit('chonKieng', IDKieng)
    document.getElementById("linkInfo").setAttribute('href', LinkKieng)
}

socket.on('deXuat', function(gender, age){
    var list, index;
    list = document.getElementsByClassName(gender + ' ' + age);
    console.log(gender + ' ' + age)
    for (index = 0; index < list.length; ++index) {
            list[index].setAttribute('style', 'display:block !important');
    }
})
