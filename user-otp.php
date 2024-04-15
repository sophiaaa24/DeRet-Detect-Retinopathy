<?php require_once "logic.php"; ?>
<?php 
$email = $_SESSION['email'];
if($email == false){
  header('Location: login-user.php');
}
?>
<html>
<head>
    <title>Code Verification</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
<div class="card">
        <div class="card-body">
                     <form action="user-otp.php" method="POST" autocomplete="off">
                    <h2 class="text-center">Code Verification</h2>
                    <?php 
                    if(isset($_SESSION['info'])){
                        ?>
                        <div class="alert">
                            <?php echo $_SESSION['info']; ?>
                        </div>
                        <?php
                    }
                    ?>
                    <?php
                    if(count($errors) > 0){
                        ?>
                        <div class="alert">
                            <?php
                            foreach($errors as $showerror){
                                echo $showerror;
                            }
                            ?>
                        </div>
                        <?php
                    }
                    ?>
                    <div>
                        <input class="input_css" type="number" name="otp" placeholder="Enter verification code" required>
                    </div>
                    <div>
                        <input class="submit_css" type="submit" name="check" value="Submit">
                    </div>
                </form>
            </div>
        </div>
    
</body>
</html>