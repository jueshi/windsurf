let currentDirectory = "";
let currentFile = "";
let debounceTimer;

$(document).ready(function() {
    loadSettings();
    // Initially load nothing or current directory if available?
    // User needs to input directory.
});

function loadSettings() {
    $.get('/api/settings', function(data) {
        if (data.recent_directories) {
            const select = $('#recentDirs');
            select.empty();
            select.append('<option value="">Recent...</option>');
            data.recent_directories.forEach(dir => {
                select.append(`<option value="${dir}">${dir}</option>`);
            });

            if (data.recent_directories.length > 0) {
                currentDirectory = data.recent_directories[0];
                $('#directoryInput').val(currentDirectory);
                loadFiles();
            }
        }
    });
}

function loadRecentDir(dir) {
    if (dir) {
        $('#directoryInput').val(dir);
        loadFiles();
    }
}

function loadFiles() {
    const dir = $('#directoryInput').val();
    const subfolders = $('#subfoldersCheck').is(':checked');
    const filter = $('#fileFilter').val();

    $.ajax({
        url: '/api/files',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            directory: dir,
            include_subfolders: subfolders,
            filter: filter
        }),
        success: function(response) {
            if (response.error) {
                alert(response.error);
                return;
            }
            renderFileTable(response.files);
            currentDirectory = response.current_directory;
        }
    });
}

function debouncedLoadFiles() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(loadFiles, 300);
}

function renderFileTable(files) {
    const tbody = $('#fileTable tbody');
    tbody.empty();

    files.forEach(file => {
        const tr = $('<tr>');
        tr.append(`<td>${file.Name}</td>`);
        tr.append(`<td>${file.File_Path}</td>`);
        tr.append(`<td>${file.Size}</td>`);
        tr.append(`<td>${file.Modified}</td>`);

        tr.click(function() {
            $('#fileTable tr').removeClass('selected');
            $(this).addClass('selected');
            loadCsv(file.File_Path);
        });

        tbody.append(tr);
    });
}

function loadCsv(filepath) {
    if (filepath) currentFile = filepath;
    if (!currentFile) return;

    const rowFilter = $('#rowFilter').val();
    const colFilter = $('#colFilter').val();

    $('#csvStatus').text("Loading...");

    $.ajax({
        url: '/api/csv',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            filepath: currentFile,
            row_filter: rowFilter,
            col_filter: colFilter
        }),
        success: function(response) {
            if (response.error) {
                $('#csvStatus').text("Error: " + response.error);
                return;
            }
            renderCsvTable(response);
        }
    });
}

function debouncedLoadCsv() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => loadCsv(), 500); // Slower debounce for CSV
}

function renderCsvTable(data) {
    const thead = $('#csvTable thead');
    const tbody = $('#csvTable tbody');
    thead.empty();
    tbody.empty();

    if (data.columns && data.columns.length > 0) {
        const trHead = $('<tr>');
        data.columns.forEach(col => {
            trHead.append(`<th>${col}</th>`);
        });
        thead.append(trHead);

        data.data.forEach(row => {
            const tr = $('<tr>');
            data.columns.forEach(col => {
                tr.append(`<td>${row[col] !== null ? row[col] : ''}</td>`);
            });
            tbody.append(tr);
        });
    }

    let status = `${data.total_rows} rows`;
    if (data.truncated) status += " (truncated)";
    $('#csvStatus').text(status);
}

function refreshFiles() {
    loadFiles();
    if (currentFile) loadCsv();
}

function deleteSelected() {
    const selectedRow = $('#fileTable tr.selected');
    if (selectedRow.length === 0) return;

    // Naively assume the path is in the second cell
    const path = selectedRow.find('td:nth-child(2)').text();

    if (confirm(`Delete ${path}?`)) {
        $.ajax({
            url: '/api/action',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                type: 'delete',
                files: [path]
            }),
            success: function(response) {
                if (response.error) alert(response.error);
                else {
                    refreshFiles();
                    $('#csvTable thead').empty();
                    $('#csvTable tbody').empty();
                    currentFile = "";
                }
            }
        });
    }
}

// Correlation Analysis Functions
function showCorrelationDialog() {
    if (!currentFile) {
        alert("Please select a file first");
        return;
    }

    // Get numeric columns
    $.ajax({
        url: '/api/columns/numeric',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            filepath: currentFile
        }),
        success: function(response) {
            if (response.error) {
                alert(response.error);
                return;
            }
            if (!response.numeric_columns || response.numeric_columns.length === 0) {
                alert("No numeric columns found in this file.");
                return;
            }

            const select = $('#targetColumn');
            select.empty();
            response.numeric_columns.forEach(col => {
                select.append(`<option value="${col}">${col}</option>`);
            });

            $('#plotContainer').empty();
            $('#correlationModal').css('display', 'block');
        }
    });
}

function closeCorrelationModal() {
    $('#correlationModal').css('display', 'none');
}

function runCorrelation() {
    const targetCol = $('#targetColumn').val();
    const topN = $('#topN').val();

    if (!targetCol) return;

    $('#plotContainer').html('Loading plots...');

    $.ajax({
        url: '/api/plot/correlation',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            filepath: currentFile,
            target_column: targetCol,
            top_n: topN
        }),
        success: function(response) {
            if (response.error) {
                $('#plotContainer').html("Error: " + response.error);
                return;
            }

            $('#plotContainer').empty();
            if (response.plots && response.plots.length > 0) {
                response.plots.forEach(plot => {
                    const div = $('<div class="plot-item">');
                    div.append(`<img src="data:image/png;base64,${plot.image}">`);
                    div.append(`<p>${plot.column} (Corr: ${plot.correlation.toFixed(3)})</p>`);
                    $('#plotContainer').append(div);
                });
            } else {
                $('#plotContainer').html("No correlations found.");
            }
        }
    });
}
