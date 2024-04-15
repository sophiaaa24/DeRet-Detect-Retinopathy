<?php require_once "logic.php"; ?>
<?php
if($_SESSION['info'] == false){
    header('Location: login-user.php');  
}
?>
<html>
<head>
     <title>Login Form</title>
    <link rel="stylesheet" href="css/styles.css">
    
</head>
<body>
    <div class="card">
        <div class="card-body">
            <div>
            <?php 
            if(isset($_SESSION['info'])){
                ?>
                <div class="alert">
                <?php echo $_SESSION['info']; ?>
                </div>
                <?php
            }
            ?>
                <form action="login-user.php" method="POST">
                    <div class="card-body">
                        <input class="submit_css" type="submit" name="login-now" value="Login Now">
                    </div>
                </form>
            </div>
        </div>
    </div>
    
</body>
</html>
