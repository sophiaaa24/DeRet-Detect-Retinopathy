<?php require_once "logic.php"; ?>
<html>
<head>
    <title>Forgot Password</title>
     <link rel="stylesheet" href="css/styles.css">
</head>
<body>
<body>
    <div class="card">
        <div class="card-body">
            <h2 class="title">Forgot Password</h2>
            <form action="forgot-password.php" method="POST">
                <?php if(count($errors) > 0): ?>
                    <div class="alert">
                        <?php foreach($errors as $error): ?>
                        <?php echo $error; ?>
                        <?php endforeach; ?>
                    </div>
                <?php endif; ?>
                <div>
                    <input class="input_css" type="email" name="email" placeholder="Enter email address" required value="<?php echo $email ?>">
                </div>
                <div>
                    <input class="submit_css" type="submit" name="check-email" value="Continue">
                </div>
            </form>
        </div>
    </div>
</body>
</html>