<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Organization</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            height: 100%;
        }

        body {
            font-family: Arial, sans-serif;
            color: #fff; 
        }

        header {
            background-color:black; 
          color: white;
            text-align: center;
            padding: 1em 0;
        }

        .background-container {
            background-image: url('img/homepage.jpg'); 
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 130vh; 
        }

        nav {
            background-color: rgba(0, 0, 0, 0.6); 
            overflow: hidden;
	text-align: right;
        }

        nav a {
            display: inline-block;
            color: white;
            text-align: right;
            padding: 14px 16px;
            text-decoration: none;
            margin-right: 15px;
	    font-weight: bold;
        }

        nav a:hover {
            background-color: #ddd;
            color: black;
        }

        section {
            padding: 20px;
            text-align: center;
        }

        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
        }
    </style>
</head>
<body>

    <header>
        <h1>DeRet: Detect Retinopathy</h1>
    </header>

    <div class="background-container">
        <nav>
            <a href="">Home</a>
            <a href="service.html">Services</a>
            <a href="https://mediafiles.botpress.cloud/5e3344f9-1c64-4724-a122-2fdd264c79a7/webchat/bot.html">FAQ</a>
	    <a href="login-user.php">Login</a>
        </nav>
    </div>

</body>
</html>
