const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');
//const path = require('path');
const app = express();
//app.use(express.static(path.join(__dirname, 'build')));

filePath = "../../resources/replays/temp.txt";

app.get('/temp', function (req, res) {
    res.setHeader('Content-Type', 'application/json');
    return res.send(fs.readFileSync(filePath, {encoding: 'utf-8'}));
});

app.get('/', (req, res) => {
    return res.send({"body": "hello", "statusCode": 200})
});

app.listen(process.env.PORT || 8080);