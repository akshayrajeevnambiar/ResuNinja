submitFormOption = function(submitFormOptionValue) {
    var submitFeedbackForm = document.getElementById('submitPredictionFeedback');
    submitFeedbackForm.style.display = submitFormOptionValue ? "block" : "none";
}

submitFeedback = function(feedbackSubmitted) {
    var feedbackSubmissionConfirmation = document.getElementById('submitFeedbackButtonSubmitted');
    feedbackSubmissionConfirmation.style.display = "block";
    
    var submitFeedbackFormElements = document.getElementById('submitPredictionFeedback').elements;
    var radioElements = document.getElementsByClassName('submitFeedbackRadioOptions');
    for (var i = 0, len = radioElements.length; i < len; ++i) {
        radioElements[i].disabled = true;
    }

    for (var i = 0, len = submitFeedbackFormElements.length; i < len; ++i) {
        submitFeedbackFormElements[i].disabled = true;
        submitFeedbackFormElements[i].style.color = 'lightgrey';
    }

    var v = document.getElementById('submitFeedbackContainer');
    v.style.color = 'lightgrey';

    var feedbackResumes = document.getElementById('submitPredictionFeedback').getElementsByClassName('feedbackRow');
    for (var i = 0, len = feedbackResumes.length; i < len; ++i) {
        feedbackResumes[i].getElementsByTagName('label')[0].style.color = 'lightgrey';
    }
}

updateList = function () {
    var input = document.getElementById('fileInput');
    var output = document.getElementById('fileList');
    var errorContainer = document.getElementById('errorContainer');
    var children = "";
    for (var i = 0; i < input.files.length; ++i) {
        children += '<li>' + input.files.item(i).name + '</li>';
    }
    output.innerHTML = '<ul>' + children + '</ul>';

    var uploadedFileViewer = document.getElementById('uploadedFiles');

    uploadedFileViewer.style.display = input.files.length > 0 ? "block" : "none";
    errorContainer.style.display = input.files.length > 0 ? "none" : "block";
}

validateInput = function () {
    try {

        var input = document.getElementById('fileInput');
        var inputArray = Array.from(input.files)

        var vaidFiles = inputArray.filter(x => 
            x.type == "text/plain" || 
            x.type == "application/pdf" || 
            x.type.includes('image/') || 
            x.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        if (vaidFiles.length <= 0) {
            var output = document.getElementById('errorContainer');
            var children = "";
            children += '<li>Please Upload valid files!</li>';
            output.innerHTML = '<ul>' + children + '</ul>';
            output.style.display = "block";
            return false;
        }

        return true;
    }
    catch (error) {
        var output = document.getElementById('errorContainer');
        var children = "";
        children += '<li>An unexpected error has occurred! Please try again later.</li>';
        output.innerHTML = '<ul>' + children + '</ul>';
        output.style.display = "block";
        return false;
    }
}