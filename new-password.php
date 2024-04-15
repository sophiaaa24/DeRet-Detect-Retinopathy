<?php require_once "logic.php"; ?>
<html>
<head>
    <title>Forgot Password</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
    <<div class="card">
        <div class="card-body">
            <div>
                <form action="new-password.php" method="POST" autocomplete="off">
                    <h2 class="text-center">New Password</h2>
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
                        <input class="input_css" type="password" name="password" placeholder="Create New Password" pattern="^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$" required>
                    </div>
                    <div>
                        <input class="input_css" type="password" name="cpassword" placeholder="Confirm Your Password" pattern="^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$" required>
                    </div>
                    <div>
                        <input class="submit_css" type="submit" name="change-password" value="Change">
                    </div>
                </form>
            </div>
        </div>
    </div>
    
</body>
</html>